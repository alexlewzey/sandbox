from faker import Faker


class Site:
    fake = Faker(['en_GB'])
    url = 'original'

    def __init__(self):
        self._emails = {}
        self.e1 = self.fake.email()
        self.e2 = self.fake.email()

    def __setattr__(self, k, v):
        if not k.startswith('_'): self._emails[k] = v
        super().__setattr__(k, v)

    def emails(self):
        for email in self._emails.values():
            yield email


site = Site()

print(site.url)

print(site.url)
print(Site.url)

Site.url = 'new change'
print(site.url, Site.url)
id(site.url) == id(Site.url)

print('All instances will use the class level variable until it is assigned at the instance level in which case you '
      'are now using a different variable, At a different memory reference.')